module     p0_ubaru_httbar_d65h9l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d65h9l1.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki
   use p0_ubaru_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_color
      use p0_ubaru_httbar_abbrevd65h9
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc65(20)
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl5
      complex(ki) :: QspQ
      Qspk2 = dotproduct(Q,k2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl5 = dotproduct(Q,l5)
      QspQ = dotproduct(Q,Q)
      acc65(1)=abb65(8)
      acc65(2)=abb65(9)
      acc65(3)=abb65(10)
      acc65(4)=abb65(11)
      acc65(5)=abb65(12)
      acc65(6)=abb65(13)
      acc65(7)=abb65(18)
      acc65(8)=abb65(19)
      acc65(9)=abb65(20)
      acc65(10)=abb65(25)
      acc65(11)=abb65(30)
      acc65(12)=acc65(3)*Qspk2
      acc65(13)=acc65(5)*Qspvak2l3
      acc65(12)=acc65(13)+acc65(12)+acc65(2)
      acc65(12)=Qspvak1k2*acc65(12)
      acc65(13)=acc65(1)*Qspk2
      acc65(14)=acc65(6)*Qspvak2l3
      acc65(15)=Qspval4l5*acc65(9)
      acc65(16)=Qspval4k2*acc65(10)
      acc65(17)=Qspvak1l5*acc65(8)
      acc65(18)=Qspvak1l3*acc65(11)
      acc65(19)=Qspl5*acc65(7)
      acc65(20)=QspQ*acc65(4)
      brack=acc65(12)+acc65(13)+acc65(14)+acc65(15)+acc65(16)+acc65(17)+acc65(1&
      &8)+acc65(19)+acc65(20)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d65h9l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1, only: epspow
      use p0_ubaru_httbar_kinematics
      use p0_ubaru_httbar_abbrevd65h9
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d65
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d65 = 0.0_ki
      d65 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d65, ki), aimag(d65), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d65h9l1
