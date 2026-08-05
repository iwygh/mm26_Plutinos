program midplane_32
implicit none
real*8, parameter::pi = 2.0d0*dasin(1.0d0)
character*500 infile_plutinos,infile_vt,outfile
character*10 des
integer nlines,nobj,iobj,maxiter,nreps,irep
real*8 idegmean,nodedegmean,theta_mid_deg,phi_mid_deg,i_mid_deg,node_mid_deg,&
    & qmean,pmean,smean,q_mid,p_mid,s_mid,dot,diff_deg,t0,t1
real*8,allocatable :: aau_list(:),e_list(:),ideg_list(:),nodedeg_list(:),& 
    & perideg_list(:),Mdeg_list(:),vtx_list(:),vty_list(:),vtz_list(:),& 
    & xau_list(:),yau_list(:),zau_list(:),rau_list(:),rpau_list(:),eclatdeg_list(:),&
    & eclondeg_list(:),vx_list(:),vy_list(:),vz_list(:),irad_list(:),noderad_list(:),&
    & perirad_list(:),Mrad_list(:),eclatrad_list(:),eclonrad_list(:),&
    & i_mid_deg_list(:),node_mid_deg_list(:),diff_deg_list(:)
character*10,allocatable :: des_list(:)
INTEGER :: i_seed
INTEGER, DIMENSION(:), ALLOCATABLE :: a_seed
INTEGER, DIMENSION(1:8) :: dt_seed
CALL RANDOM_SEED(size=i_seed)
ALLOCATE(a_seed(1:i_seed))
CALL RANDOM_SEED(get=a_seed)
CALL DATE_AND_TIME(values=dt_seed)
a_seed(i_seed)=dt_seed(8); a_seed(1)=dt_seed(8)*dt_seed(7)*dt_seed(6)
CALL RANDOM_SEED(put=a_seed)
DEALLOCATE(a_seed)
maxiter = 10000000
nreps = 3
outfile = trim(adjustl('b020_fortran_mean_planes_vm17_2026feb12_gplusminusnone_ijobNNN.txt'))
outfile = trim(adjustl(outfile))
infile_plutinos = trim(adjustl('b020_2026feb12_plutinos_fortran_gplusminusnone_2026feb12.csv'))
infile_plutinos = trim(adjustl(infile_plutinos))
infile_vt = trim(adjustl('b020_2026feb12_plutinos_vtvec_gplusminusnone_2026feb12.csv'))
infile_vt = trim(adjustl(infile_vt))
call get_line_count(infile_plutinos,nlines)
nobj = nlines-1
! write(*,*) nlines,nobj
allocate(des_list(1:nobj),aau_list(1:nobj),e_list(1:nobj),ideg_list(1:nobj),&
    & nodedeg_list(1:nobj),perideg_list(1:nobj),Mdeg_list(1:nobj),&
    & vtx_list(1:nobj),vty_list(1:nobj),vtz_list(1:nobj),xau_list(1:nobj),&
    & yau_list(1:nobj),zau_list(1:nobj),rau_list(1:nobj),rpau_list(1:nobj),&
    & eclatdeg_list(1:nobj),eclondeg_list(1:nobj),vx_list(1:nobj),vy_list(1:nobj),& 
    & vz_list(1:nobj),irad_list(1:nobj),noderad_list(1:nobj),&
    & perirad_list(1:nobj),Mrad_list(1:nobj),eclatrad_list(1:nobj),&
    & eclonrad_list(1:nobj))
allocate(i_mid_deg_list(1:nreps),node_mid_deg_list(1:nreps),diff_deg_list(1:nreps))
open(99,file=infile_plutinos)
read(99,*) ! skip first line (header)
do iobj = 1,nobj
    read(99,*) des_list(iobj),aau_list(iobj),e_list(iobj),ideg_list(iobj),& 
          & nodedeg_list(iobj),perideg_list(iobj),Mdeg_list(iobj)
end do
close(99)
open(99,file=infile_vt)
read(99,*) ! skip first line (header)
do iobj = 1,nobj
    read(99,*) des,vtx_list(iobj),vty_list(iobj),vtz_list(iobj)
end do
close(99)
do iobj = 1,nobj
    call radians(ideg_list(iobj),irad_list(iobj))
    call radians(nodedeg_list(iobj),noderad_list(iobj))
    call radians(perideg_list(iobj),perirad_list(iobj))
    call radians(Mdeg_list(iobj),Mrad_list(iobj))
end do
do iobj = 1,nobj
    call el2xv(1.0d0,aau_list(iobj),e_list(iobj),irad_list(iobj),noderad_list(iobj),&
        & perirad_list(iobj),Mrad_list(iobj),xau_list(iobj),yau_list(iobj),&
        & zau_list(iobj),vx_list(iobj),vy_list(iobj),vz_list(iobj))
    rau_list(iobj) = dsqrt(xau_list(iobj)**2+yau_list(iobj)**2+zau_list(iobj)**2)
    rpau_list(iobj) = dsqrt(xau_list(iobj)**2+yau_list(iobj)**2)
    eclatrad_list(iobj) = datan2(zau_list(iobj),rpau_list(iobj))
    call degrees(eclatrad_list(iobj),eclatdeg_list(iobj))
    eclonrad_list(iobj) = datan2(yau_list(iobj),xau_list(iobj))
    call degrees(eclonrad_list(iobj),eclondeg_list(iobj))
end do
call fit_plane_vtxyz(nobj,vtx_list,vty_list,vtz_list, & ! inputs
                    & theta_mid_deg,phi_mid_deg,idegmean,nodedegmean) ! outputs
open(99,file=outfile)
write(99,'(A21,f10.2,f10.2)') 'idegmean, nodedegmean',idegmean,nodedegmean
write(99,'(A63)')'irep, nreps, i_mid_deg, node_mid_deg, diff_deg, elapsed_minutes'
write(*,'(A21,f10.2,f10.2)') 'idegmean, nodedegmean',idegmean,nodedegmean
write(*,'(A63)')'irep, nreps, i_mid_deg, node_mid_deg, diff_deg, elapsed_minutes'
call cpu_time(t0)
do irep = 1,nreps
!     write(*,'(A25,I4,I4)') 'irep,nreps',irep,nreps
    call vm17(nobj,maxiter,des_list,irad_list,e_list,aau_list,idegmean,nodedegmean,&
        & eclatrad_list,eclonrad_list,i_mid_deg,node_mid_deg)
!     write(*,'(A25,f10.2,f10.2)') 'i_mid_deg,node_mid_deg',i_mid_deg,node_mid_deg
    qmean = dsin(idegmean*pi/180.0d0)*dcos(nodedegmean*pi/180.0d0)
    pmean = dsin(idegmean*pi/180.0d0)*dsin(nodedegmean*pi/180.0d0)
    smean = dcos(idegmean*pi/180.0d0)
    q_mid = dsin(i_mid_deg*pi/180.0d0)*dcos(node_mid_deg*pi/180.0d0)
    p_mid = dsin(i_mid_deg*pi/180.0d0)*dsin(node_mid_deg*pi/180.0d0)
    s_mid = dcos(i_mid_deg*pi/180.0d0)
    dot = qmean*q_mid + pmean*p_mid + smean*s_mid
    diff_deg = dacos(dot) * 180.0d0/pi
!     write(*,'(A25,f10.2)') 'diff_deg',diff_deg
    i_mid_deg_list(irep) = i_mid_deg
    node_mid_deg_list(irep) = node_mid_deg
    diff_deg_list(irep) = diff_deg
    call cpu_time(t1)
!     write(*,'(A25,f10.2)') 'elapsed minutes',(t1-t0)/60.0d0
    write(*,'(I4,I4,f6.2,f7.2,f6.2,f11.2)') irep,nreps,i_mid_deg,node_mid_deg,&
        & diff_deg,(t1-t0)/60.0d0
    write(99,'(I4,I4,f6.2,f7.2,f6.2,f11.2)') irep,nreps,i_mid_deg,node_mid_deg,&
        & diff_deg,(t1-t0)/60.0d0
end do
close(99)
end program

subroutine vm17(nobj,maxiter,name_list,irad_list,e_list,aau_list,idegmean,nodedegmean,&
    & eclat_list,eclon_list,i_mid_deg,node_mid_deg)
implicit none
real*8, parameter::pi = 2.0d0*dasin(1.0d0)
real*8 sigdeg1,sigdeg2,GM,idegmean,nodedegmean,q0,p0,inc,temp,temp2,ideg_threshold,&
    & temp_threshold,w,sigma,node,argperi,meananom,et,at,i_mid_deg,node_mid_deg,diff_lat,diff_lon,&
    & eccolat,eclon,eclat,x,y,z,vx,vy,vz,eclat_comp,eclon_comp
real*8 vtx_list(nobj),vty_list(nobj),vtz_list(nobj),&
    & irad_list(nobj),e_list(nobj),aau_list(nobj),eclat_list(nobj),eclon_list(nobj)
character*10 name_obj
character*10 name_list(nobj)
integer nobj,maxiter,cnt,match,jobj,iobj
q0 = dsin(idegmean*pi/180.0d0)*dcos(nodedegmean*pi/180.0d0)
p0 = dsin(idegmean*pi/180.0d0)*dsin(nodedegmean*pi/180.0d0)
sigma = dsin(10.4d0 * pi/180.0d0)
do iobj = 1,nobj
    cnt = 0
    match = 0
    name_obj = name_list(iobj)
    do while ((match .eq. 0) .and. (cnt .lt. maxiter+3))
        cnt = cnt + 1
        if (cnt .eq. (maxiter-1) ) then
            write(*,*) 'maxiter iobj',iobj,nobj,name_obj
        end if
        inc = 10.0d0
        do while (inc .gt. 1.0d0)
            w = 1.5d0
            do while (w .gt. 1.0d0)
                call random_number(temp)
                call random_number(temp2)
                w = (2.0d0*temp-1.0d0)**2 + (2.0d0*temp2-1.0d0)**2
            end do
            temp = (2.0d0*temp-1.0d0) * dsqrt((-2.0d0*dlog(w))/w)
            temp2 = (2.0d0*temp2-1.0d0) * dsqrt((-2.0d0*dlog(w))/w)
            inc = dsqrt((q0+temp*sigma)**2+(p0+temp2*sigma)**2)
        end do
        inc = dasin(inc)
        node = modulo(datan2(p0+temp2*sigma,q0+temp*sigma),2.0d0*pi)
        call random_number(temp)
        argperi = temp*2.0d0*pi
        call random_number(temp)
        meananom = temp*2.0d0*pi
        jobj = 0 ! randomly select one of the observed objects
        do while (jobj .eq. 0 .or. jobj .gt. nobj)
            call random_number(temp) ! random real number between 0 and 1
                jobj = nint(temp*nobj+0.5d0)
        end do
        call random_number(temp) ! random number between 0 and 1
        et = 0.95d0*e_list(jobj) + (1.05d0-0.95d0)*e_list(jobj)*temp
        call random_number(temp) ! random number between 0 and 1
        at = 0.99d0*aau_list(jobj) + (1.01d0-0.99d0)*aau_list(jobj)*temp
        call el2xv(1.0d0,at,et,inc,node,argperi,meananom,x,y,z,vx,vy,vz)
        eclat = datan2(z,dsqrt(x*x + y*y))
        eclon = datan2(y,x)
        eclon = modulo(eclon,2.0d0*pi)
        eclat = modulo(eclat,2.0d0*pi)
        eclon_comp = modulo(eclon_list(iobj),2.0d0*pi)
        eclat_comp = modulo(eclat_list(iobj),2.0d0*pi)
        diff_lat = dabs(eclat*180.0d0/pi - eclat_comp*180.0d0/pi) ! easier to compare in degrees than radians
        diff_lon = dabs(eclon*180.0d0/pi - eclon_comp*180.0d0/pi) ! easier to compare in degrees than radians
        temp = dabs(diff_lon-360.0d0)
        if (temp .lt. diff_lon) diff_lon = temp ! a difference of 359 degrees is the same size as a difference of 1 degree
        if ( (diff_lon .lt. 5.0d0) .and. (diff_lat .lt. 1.0d0) ) match = 1
    end do
    eccolat = pi/2.0d0 - eclat
    vtx_list(iobj) = -dcos(inc)*dsin(eccolat)*dsin(eclon) - dsin(inc)*dcos(node)*dcos(eccolat)
    vty_list(iobj) = dcos(inc)*dsin(eccolat)*dcos(eclon) - dsin(inc)*dsin(node)*dcos(eccolat)
    vtz_list(iobj) = dsin(inc)*dsin(node)*dsin(eccolat)*dsin(eclon) + dsin(inc)*dcos(node)*dsin(eccolat)*dcos(eclon)
end do
call fit_plane_vtxyz(nobj,vtx_list,vty_list,vtz_list,temp,temp2,i_mid_deg,node_mid_deg)
return
end subroutine

subroutine fit_plane_vtxyz(nobj,vtx_list,vty_list,vtz_list, & ! inputs
                    & theta_mid_deg,phi_mid_deg,i_mid_deg,node_mid_deg) ! outputs
implicit none
integer iobj,nobj
real*8 thetamin_deg,thetamax_deg,dtheta_deg,phimin_deg,phimax_deg,dphi_deg
real*8 thetamin,thetamax,dtheta,phimin,phimax,dphi,theta_mid_deg,phi_mid_deg
real*8 sm,smmin,theta_mid,phi_mid,delta,nx,ny,nz,theta,phi,i_mid_deg,node_mid_deg
real*8 vtx_list(1:nobj),vty_list(1:nobj),vtz_list(1:nobj)
real*8, parameter::pi = 2.0d0*dasin(1.0d0)
thetamin_deg = 0.01d0
thetamax_deg = 10.0d0
dtheta_deg = 0.01d0
phimin_deg = 0.0d0
phimax_deg = 359.98d0
dphi_deg = 0.01d0
thetamin = thetamin_deg * pi / 180.0d0
thetamax = thetamax_deg * pi / 180.0d0
dtheta = dtheta_deg * pi / 180.0d0
phimin = phimin_deg * pi / 180.0d0
phimax = phimax_deg * pi / 180.0d0
dphi = dphi_deg * pi / 180.0d0
smmin = 1.0d9
theta_mid = 1000.0d0
phi_mid = 1000.0d0
theta = thetamin
phi = phimin
do while (theta .lt. thetamax)
    do while (phi .lt. phimax)
        nx = dsin(theta) * dcos(phi)
        ny = dsin(theta) * dsin(phi)
        nz = dcos(theta)
        sm = 0.0d0
        do iobj = 1,nobj
            delta = vtx_list(iobj)*nx + vty_list(iobj)*ny + vtz_list(iobj)*nz
            sm = sm + dabs(delta)
        end do
        if (sm .lt. smmin) then
            smmin = sm
            theta_mid = theta
            phi_mid = phi
        end if
        phi = phi + dphi
    end do
    theta = theta + dtheta
    phi = phimin
end do
theta_mid_deg = theta_mid * 180.0d0 / pi
phi_mid_deg = phi_mid * 180.0d0 / pi
i_mid_deg = theta_mid_deg
node_mid_deg = phi_mid_deg - 270.0d0
node_mid_deg = modulo(node_mid_deg,360.0d0)
return
end subroutine

subroutine radians(indeg,outrad)
implicit none
real*8, parameter::pi = 2.0d0*dasin(1.0d0)
real*8 indeg,outrad
outrad = indeg * pi/180.0d0
return
end subroutine

subroutine degrees(inrad,outdeg)
implicit none
real*8, parameter::pi = 2.0d0*dasin(1.0d0)
real*8 inrad,outdeg
outdeg = inrad * 180.0d0/pi
return
end subroutine


subroutine get_line_count(infile,& ! inputs
                  & line_count ) ! outputs
implicit none
integer line_count,io
character(len=*) infile
open(99,file=infile,iostat=io)
if (io/=0) stop 'get_line_count: Cannot open file!'
line_count = 0
do
    read(99,*,iostat=io)
    if (io/=0) exit
    line_count = line_count + 1
end do
close(99)
return
end subroutine

!******************************************************************************
!                             SUBROUTINE EL2XV
!******************************************************************************
! Computes cartesian positions and velocities given central mass, orbital
! elements, and ialpha
! Arguments:
!   GM: G times central mass
!   \\\\\ taken out!  now only good for ellipses
!   \\\\\\  ialpha: integer for conic section type ( =+1 for hyperbola, 0 for parabola,
!   \\\\\\\        and -1 for ellipse)
!   a : semi-major axis (or pericentric distance if a parabola),
!   e : eccentricity
!   inc : inclination
!   capom : longitude of ascending node , often called capital OMEGA
!   omega : argument of perihelion (from ascending node)
!   capmnq : either M, N or Q.
! ALGORITHM:  See Fitzpatrick "Principles of Cel. Mech."
! AUTHOR:  M. Duncan.
! DATE WRITTEN:  May 11, 1992.
! REVISIONS: May 26 - now use better Kepler solver for ellipses
!   and hyperbolae called EHYBRID.F and FHYBRID.F
! Last change: 19 Feb 1998 // altered Dec 14,2009
!******************************************************************************
subroutine el2xv(GM,a,e,inc,capom,omega,capmnq,x,y,z,vx,vy,vz)
implicit none
!..............................................................................
! arguments
real*8 GM,a,e,inc,capom,omega,capmnq
real*8 x,y,z,vx,vy,vz
!..............................................................................
! internal variables
real*8 ehybrid,cape,fhybrid,capf,zget,zpara
real*8 sp,cp,so,co,si,ci
real*8 d11,d12,d13,d21,d22,d23
real*8 scap,ccap,shcap,chcap
real*8 sqe,sqgma,xfac1,xfac2,ri,vfac1,vfac2
!------------------------------------------------------------------------------
! Generate rotation matrices (on p. 42 of Fitzpatrick)
call scget(omega,sp,cp)
call scget(capom,so,co)
call scget(inc,si,ci)
d11 = cp*co - sp*so*ci
d12 = cp*so + sp*co*ci
d13 = sp*si
d21 = -sp*co - cp*so*ci
d22 = -sp*so + cp*co*ci
d23 = cp*si
! Get the other quantities depending on orbit type ( i.e. IALPHA)
! only valid for ellipses
cape = ehybrid(e,capmnq)
call scget(cape,scap,ccap)
sqe = dsqrt(1.d0 -e*e)
sqgma = dsqrt(GM*a)
xfac1 = a*(ccap - e)
xfac2 = a*sqe*scap
ri = 1.d0/(a*(1.d0 - e*ccap))
vfac1 = -ri * sqgma * scap
vfac2 = ri * sqgma * sqe * ccap
x =  d11*xfac1 + d21*xfac2
y =  d12*xfac1 + d22*xfac2
z =  d13*xfac1 + d23*xfac2
vx = d11*vfac1 + d21*vfac2
vy = d12*vfac1 + d22*vfac2
vz = d13*vfac1 + d23*vfac2
return
end subroutine

!******************************************************************************
real*8 FUNCTION EHYBRID(E,M)
!******************************************************************************
!       PURPOSE:  Solves Kepler's eqn.
!       ARGUMENTS:  Input is eccentricity E and mean anomaly M.
!       RETURNS the eccentric anomaly EHYBRID
!       ALGORITHM: For e < 0.18 uses fast routine ESOLMD
!	         For larger e but less than 0.8, uses EGET
!	         For e > 0.8 uses EHIE
!       REMARKS: Only EHIE brings M and E into range (0,TWOPI)
!       AUTHOR: M. Duncan
!       DATE WRITTEN: May 25,1992.
!******************************************************************************
implicit none
real*8 e,m,esolmd,eget,ehie
if(e .lt. 0.18d0) then
  EHYBRID = esolmd(e,m)
else
  if( e .le. 0.8d0) then
     EHYBRID = eget(e,m)
  else
     EHYBRID = ehie(e,m)
  endif
endif
return
end function

!******************************************************************************
real*8 FUNCTION ESOLMD(E,M)
!******************************************************************************
!       PURPOSE:  Solves Kepler's eqn.
!       ARGUMENTS:  Input is eccentricity E and mean anomaly EM.
!       RETURNs the eccentric anomaly ESOLVE
!       ALGORITHM: Some sort of quartic convergence from Wisdom.
!       REMARKS: Only good for small eccentricity since it only
!         iterates once. (good for planet orbits)
!      	  also does not put M between 0. and 2*pi
!       INCLUDES: needs SCGET.F
!       AUTHOR: M. Duncan
!       DATE WRITTEN: May 7, 1992.
!******************************************************************************
implicit none
real*8 x,e,m,sm,cm,sx,cx
real*8 es,ec,f,fp,fpp,fppp,dx
call scget(m,sm,cm)
x = m + e*sm*( 1.d0 + e*( cm + e*( 1.d0 -1.5d0*sm*sm)))
call scget(x,sx,cx)
es = e*sx
ec = e*cx
f = x - es  - m
fp = 1.d0 - ec
fpp = es
fppp = ec
dx = -f/fp
dx = -f/(fp + dx*fpp/2.d0)
dx = -f/(fp + dx*fpp/2.d0 + dx*dx*fppp/6.d0)
esolmd = x + dx
return
end function

!******************************************************************************
real*8 FUNCTION EGET(E,M)
!******************************************************************************
!       PURPOSE:  Solves Kepler's eqn.
!       ARGUMENTS:  Input is eccentricity E and mean anomaly M.
!       RETURNs the eccentric anomaly EGET
!       ALGORITHM: Quartic convergence from Danby
!       REMARKS: For results very near roundoff, give it M between
!           0 and 2*pi. One can condition M before calling EGET
!           by calling my double precision function MOD2PI(M).
!           This is not done within the routine to speed it up
!           and because it works fine even for large M.
!       AUTHOR: M. Duncan
!       DATE WRITTEN: May 7, 1992.
!       REVISIONS: May 21, 1992.  Now have it go through EXACTLY two
!           iterations with the premise that it will only be called if
!           we have an ellipse with e between 0.15 and 0.8
!******************************************************************************
! MAY 21 : FOR e < 0.18 use ESOLMD for speed and sufficient accuracy
! MAY 21 : FOR e > 0.8 use EHIE - this one may not converge fast enough.
implicit none
real*8 x,e,m,sm,cm,sx,cx
real*8 es,ec,f,fp,fpp,fppp,dx
call scget(m,sm,cm)
! begin with a guess accurate to order ecc**3
x = m + e*sm*( 1.d0 + e*( cm + e*( 1.d0 -1.5d0*sm*sm)))
! go through one iteration for improved estimate
call scget(x,sx,cx)
es = e*sx
ec = e*cx
f = x - es  - m
fp = 1.d0 - ec
fpp = es
fppp = ec
dx = -f/fp
dx = -f/(fp + dx*fpp/2.d0)
dx = -f/(fp + dx*fpp/2.d0 + dx*dx*fppp/6.d0)
eget = x + dx
! Do another iteration.
! For m between 0 and 2*pi this seems to be enough to get near roundoff
! error for eccentricities between 0 and 0.8
x = eget
call scget(x,sx,cx)
es = e*sx
ec = e*cx
f = x - es  - m
fp = 1.d0 - ec
fpp = es
fppp = ec
dx = -f/fp
dx = -f/(fp + dx*fpp/2.d0)
dx = -f/(fp + dx*fpp/2.d0 + dx*dx*fppp/6.d0)
eget = x + dx
return
end function

!******************************************************************************
real*8 FUNCTION EHIE(E,M)
!******************************************************************************
!       PURPOSE:  Solves Kepler's eqn.
!       ARGUMENTS:  Input is eccentricity E and mean anomaly M.
!       RETURNs the eccentric anomaly EHIE
!       ALGORITHM: Use Danby's quartic for 3 iterations.
!                Eqn. is f(x) = x - e*sin(x+M).
!                Note that E = x + M first guess is very good for e near 1.
!	         Need to first get M between 0. and PI and use symmetry
!                to return right answer if M between PI and 2PI
!       REMARKS: Modifies M so that both E and M are in range (0,TWOPI)
!       AUTHOR: M. Duncan
!       DATE WRITTEN: May 25,1992.
!******************************************************************************
implicit none
integer iflag,nper,niter
real*8 e,m
real*8 dx,x,sa,ca,esa,eca,f,fp
real*8, parameter:: PI = 3.1415926535897932d0
real*8, parameter::TWOPI = 2.d0*PI
real*8, parameter ::TOL = 3.d-15
integer, parameter::NPLMAX = 3
!	Bring M into the range (0,TWOPI), and if the result is greater than PI,
!	solve for (TWOPI - M).
iflag = 0
nper = m/TWOPI
m = m - nper*TWOPI
if (m .lt. 0.d0) m = m + TWOPI
if (m.gt.PI) then
   m = TWOPI - m
   iflag = 1
endif
!	Make a first guess that works well for e near 1.
x = (6.d0*m)**(1.d0/3.d0) - m
niter =0
!	Iteration loop
do niter =1,NPLMAX
    call scget(x + m,sa,ca)
    esa = e*sa
    eca = e*ca
    f = x - esa
    fp = 1.d0 -eca
    dx = -f/fp
    dx = -f/(fp + 0.5d0*dx*esa)
    dx = -f/(fp + 0.5d0*dx*(esa+0.3333333333333333d0*eca*dx))
!	    write(6,*) niter,m,dx
    x = x + dx
enddo
EHIE = m + x
if (iflag.eq.1) then
  EHIE = TWOPI - EHIE
  m = TWOPI - m
endif
return
end function

!******************************************************************************
!	                        SUBROUTINE SCGET
!******************************************************************************
!       PURPOSE:  Given an angle (in RADIANS), efficiently compute sin and cos.
!       ARGUMENTS:  Input is a real*8 ANGLE in radians.
!                 Returned are SX=sin(ANGLE) and CX=cos(ANGLE)
!       REMARKS: The HP 700 series won't return correct answers for sin
!         and cos if the angle is bigger than 3e7. We first reduce it
!         to the range [0,2pi) and use the sqrt rather than cos (it's faster)
!         BE SURE THE ANGLE IS IN RADIANS - NOT DEGREES!
!       AUTHOR:  M. Duncan.
!       DATE WRITTEN:  May 6, 1992.
!       Last change: 19 Feb 1998 (RM)
!******************************************************************************
subroutine scget(angle,sx,cx)
implicit none
integer nper
real*8 angle,sx,cx
real*8 x
real*8, parameter:: PI = 3.1415926535897932d0
real*8, parameter:: TWOPI = 2.d0*PI
real*8, parameter:: PIBY2 = 0.5d0*PI
real*8, parameter:: PI3BY2 = 1.5d0*PI
nper = angle/TWOPI
x = angle - nper*TWOPI
if(x.lt.0.d0) x = x + TWOPI
sx = dsin(x)
cx= dsqrt(1.d0 - sx*sx)
if( x .gt. PIBY2 .and. x .lt.PI3BY2) cx = -cx
return
end subroutine

!******************************************************************************
! 			       SUBROUTINE XV2EL
!******************************************************************************
!     PURPOSE:  Given the cartesian position and velocity of an orbit,
!       compute the osculating orbital elements.
!     INPUTS: position (X,Y,Z), velocity (VX,VY,VZ), and central GM
!     RETURNED:
!       ialpha: an integer: (-1 for ellipse, 0 for parabola, +1 for hyperbola)
!       a:  semi-major axis for ellipse, pericentric distance for parabola,
!           (-2*GM/energy) for hyperbola.
!       e: eccentricity
!       inc: inclination
!       node: longitude of the ascending node
!       peri: argument of perihelion
!	capmnq: mean anomaly M for an ellipse, Q for a parabola or
!               N for a hyperbola (in Fitzpatrick's notation).
!       All angles are in radians.
!     ALGORITHM: See e.g. p.70 of Fitzpatrick's "Priciples of Cel. Mech."
!     REMARKS:
!     1. If the inclination INC is less than TINY, we arbitrarily choose the
!        longitude of the ascending node NODE to be 0.0 (so the ascending
!        node is then along the X axis).
!     2. If the eccentricity, E, is less than SQRT(TINY), we arbitrarily
!        choose the argument of perihelion PERI to be 0.
!     AUTHOR: M. Duncan; May 8,1992.
! Last change: 28 Jan 1998 (RM)
!******************************************************************************
subroutine xv2el(GM,x,y,z,vx,vy,vz,a,e,inc,node,peri,capmnq)
implicit none
integer ialpha
real*8 x,y,z,vx,vy,vz
real*8 GM,a,e,inc,node,peri,capmnq
real*8 hx,hy,hz,h2,h,r,v,v2,vdotr,energy
real*8 es,ec,cw,sw,w,u
real*8 fac,cape,capf,tmpf
real*8, PARAMETER:: tiny = 2.D-15
real*8, parameter:: pi=3.1415926535897932d0,tpi=2.d0*pi
! Compute the angular momentum H, and thereby the inclination INC.
hx = y*vz - z*vy
hy = z*vx - x*vz
hz = x*vy - y*vx
h2 = hx*hx + hy*hy +hz*hz
h  = dsqrt(h2)
inc = dacos(hz/h)
! Compute longitude of ascending node CAPOM and the argument of latitude u
fac = dsqrt(hx*hx + hy*hy)/h
if(fac.lt. TINY ) then
  node = 0.d0
  u = datan2(y,x)
  if(dabs(inc - pi).lt. 10.d0*TINY) u = -u
else
  node = datan2(hx,-hy)
  u = datan2(z/dsin(inc), x*dcos(node) + y*dsin(node))
endif
if(node .lt. 0.d0) node = node + tpi
if(u .lt. 0.d0) u = u + tpi
!  Compute the radius R and velocity squared V2, and the dot
!  product RDOTV, the energy per unit mass ENERGY .
r = dsqrt(x*x + y*y + z*z)
v2 = vx*vx + vy*vy + vz*vz
v = dsqrt(v2)
vdotr = x*vx + y*vy + z*vz
energy = 0.5d0*v2 - GM/r
! ELLIPSE
  a = -0.5d0*GM/energy
  fac = 1.d0 - h2/(GM*a)
  if (fac .gt. TINY) then
     e = dsqrt ( fac ) ! eccentricity
     ec = 1.d0-r/a ! cos(eccentric anomaly)
     es = vdotr/dsqrt(GM*a) ! sin(eccentric anomaly)
     cape = datan2(es,ec) ! eccentric anomaly, E
     cw = (ec/e -e)/(1.d0 - ec)          ! cos(true anomaly)
     sw = dsqrt(1.d0-e*e)*(es/e)/(1.d0-ec) ! sin(true anomaly)
     w = datan2(sw,cw) ! true anomaly
     if(w .lt. 0.d0) w = w + tpi
     else
       e = 0.d0
       cape = u
       es = 0.d0
       w = u
     endif
     capmnq = cape - es ! mean anomaly
     peri = u - w ! arg of periapse
     peri = dmod(peri,tpi)
     if(peri .lt. 0.d0) peri = peri + tpi
return
end subroutine
