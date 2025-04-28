module     p0_ubaru_httbar_d77h1l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity1d77h1l1_qp.f90
   ! generator: buildfortran.py
   use p0_ubaru_httbar_config, only: ki => ki_qp
   use p0_ubaru_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p0_ubaru_httbar_model_qp
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_color_qp
      use p0_ubaru_httbar_abbrevd77h1_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc77(18)
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: QspQ
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      QspQ = dotproduct(Q,Q)
      acc77(1)=abb77(10)
      acc77(2)=abb77(11)
      acc77(3)=abb77(12)
      acc77(4)=abb77(13)
      acc77(5)=abb77(15)
      acc77(6)=abb77(16)
      acc77(7)=abb77(18)
      acc77(8)=abb77(19)
      acc77(9)=abb77(20)
      acc77(10)=abb77(21)
      acc77(11)=abb77(22)
      acc77(12)=abb77(24)
      acc77(13)=abb77(26)
      acc77(14)=acc77(7)*Qspval5k2
      acc77(15)=acc77(10)*Qspval4k2
      acc77(16)=Qspval5l3*acc77(5)
      acc77(17)=Qspval4l3*acc77(9)
      acc77(18)=Qspval3k2*acc77(1)
      acc77(14)=acc77(18)+acc77(17)+acc77(16)+acc77(15)+acc77(14)+acc77(3)
      acc77(14)=Qspvak1k2*acc77(14)
      acc77(15)=acc77(6)*Qspval4k2
      acc77(16)=acc77(13)*Qspval5k2
      acc77(15)=acc77(16)+acc77(15)+acc77(2)
      acc77(15)=Qspvak1l3*acc77(15)
      acc77(16)=acc77(11)*Qspval5k2
      acc77(17)=acc77(12)*Qspval4k2
      acc77(18)=QspQ*acc77(4)
      brack=acc77(8)+acc77(14)+acc77(15)+acc77(16)+acc77(17)+acc77(18)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d77h1l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd77h1_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d77
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d77 = 0.0_ki
      d77 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d77, ki), aimag(d77), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d77h1l1_qp
