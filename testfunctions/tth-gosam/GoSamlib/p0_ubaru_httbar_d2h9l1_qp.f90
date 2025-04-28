module     p0_ubaru_httbar_d2h9l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity9d2h9l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd2h9_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc2(33)
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspk2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspl5
      complex(ki) :: Qspk1
      complex(ki) :: QspQ
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspk2 = dotproduct(Q,k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspl5 = dotproduct(Q,l5)
      Qspk1 = dotproduct(Q,k1)
      QspQ = dotproduct(Q,Q)
      acc2(1)=abb2(9)
      acc2(2)=abb2(10)
      acc2(3)=abb2(11)
      acc2(4)=abb2(12)
      acc2(5)=abb2(13)
      acc2(6)=abb2(14)
      acc2(7)=abb2(15)
      acc2(8)=abb2(16)
      acc2(9)=abb2(17)
      acc2(10)=abb2(18)
      acc2(11)=abb2(23)
      acc2(12)=abb2(24)
      acc2(13)=abb2(31)
      acc2(14)=abb2(36)
      acc2(15)=abb2(39)
      acc2(16)=abb2(42)
      acc2(17)=abb2(45)
      acc2(18)=abb2(46)
      acc2(19)=abb2(47)
      acc2(20)=acc2(3)*Qspval3l5
      acc2(21)=acc2(4)*Qspk2
      acc2(22)=acc2(5)*Qspval4l5
      acc2(23)=Qspvak2l3*acc2(8)
      acc2(20)=acc2(23)+acc2(22)+acc2(21)+acc2(20)+acc2(1)
      acc2(20)=Qspvak1k2*acc2(20)
      acc2(21)=acc2(9)*Qspval4l5
      acc2(22)=acc2(11)*Qspk2
      acc2(23)=acc2(15)*Qspval3l5
      acc2(24)=-Qspval5l3*acc2(17)
      acc2(25)=Qspval5k2*acc2(13)
      acc2(26)=Qspval4k2*acc2(14)
      acc2(27)=Qspval3k2*acc2(16)
      acc2(28)=Qspvak2l5*acc2(19)
      acc2(29)=Qspvak1l5*acc2(18)
      acc2(30)=Qspvak1l3*acc2(6)
      acc2(31)=Qspl5*acc2(7)
      acc2(32)=Qspk1*acc2(10)
      acc2(33)=QspQ*acc2(12)
      brack=acc2(2)+acc2(20)+acc2(21)+acc2(22)+acc2(23)+acc2(24)+acc2(25)+acc2(&
      &26)+acc2(27)+acc2(28)+acc2(29)+acc2(30)+acc2(31)+acc2(32)+acc2(33)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d2h9l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd2h9_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d2
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d2 = 0.0_ki
      d2 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d2, ki), aimag(d2), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d2h9l1_qp
