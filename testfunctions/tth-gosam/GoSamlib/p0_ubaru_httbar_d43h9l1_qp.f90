module     p0_ubaru_httbar_d43h9l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d43h9l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd43h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc43(21)
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: QspQ
      Qspk2 = dotproduct(Q,k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      QspQ = dotproduct(Q,Q)
      acc43(1)=abb43(10)
      acc43(2)=abb43(11)
      acc43(3)=abb43(12)
      acc43(4)=abb43(13)
      acc43(5)=abb43(14)
      acc43(6)=abb43(15)
      acc43(7)=abb43(16)
      acc43(8)=abb43(17)
      acc43(9)=abb43(18)
      acc43(10)=abb43(19)
      acc43(11)=abb43(20)
      acc43(12)=abb43(21)
      acc43(13)=abb43(24)
      acc43(14)=abb43(28)
      acc43(15)=acc43(3)*Qspk2
      acc43(16)=acc43(6)*Qspval3k2
      acc43(17)=Qspval4l5*acc43(4)
      acc43(18)=Qspval4l3*acc43(5)
      acc43(19)=Qspval3l5*acc43(7)
      acc43(20)=Qspvak2l3*acc43(9)
      acc43(15)=acc43(20)+acc43(19)+acc43(18)+acc43(17)+acc43(11)+acc43(16)+acc&
      &43(15)
      acc43(15)=Qspvak1k2*acc43(15)
      acc43(16)=acc43(10)*Qspk2
      acc43(17)=acc43(14)*Qspval3k2
      acc43(18)=Qspval4k2*acc43(1)
      acc43(19)=Qspvak1l5*acc43(8)
      acc43(20)=Qspvak1l3*acc43(13)
      acc43(21)=QspQ*acc43(12)
      brack=acc43(2)+acc43(15)+acc43(16)+acc43(17)+acc43(18)+acc43(19)+acc43(20&
      &)+acc43(21)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d43h9l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd43h9_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d43
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d43 = 0.0_ki
      d43 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d43, ki), aimag(d43), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d43h9l1_qp
