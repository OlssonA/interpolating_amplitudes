module     p0_ubaru_httbar_d84h10l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p0_ubaru_httbar/helicity10d84h10l1_qp.f90
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
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc84(25)
      complex(ki) :: Qspvak1k2
      complex(ki) :: QspQ
      complex(ki) :: Qspk2
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspl4
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      QspQ = dotproduct(Q,Q)
      Qspk2 = dotproduct(Q,k2)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspl4 = dotproduct(Q,l4)
      acc84(1)=abb84(6)
      acc84(2)=abb84(7)
      acc84(3)=abb84(8)
      acc84(4)=abb84(9)
      acc84(5)=abb84(10)
      acc84(6)=abb84(13)
      acc84(7)=abb84(14)
      acc84(8)=abb84(15)
      acc84(9)=abb84(17)
      acc84(10)=abb84(22)
      acc84(11)=abb84(23)
      acc84(12)=abb84(24)
      acc84(13)=abb84(25)
      acc84(14)=abb84(26)
      acc84(15)=abb84(27)
      acc84(16)=acc84(15)*Qspvak1k2
      acc84(17)=QspQ*acc84(10)
      acc84(18)=Qspk2*acc84(1)
      acc84(16)=acc84(18)+acc84(17)+acc84(4)+acc84(16)
      acc84(16)=Qspvak2k1*acc84(16)
      acc84(17)=QspQ*acc84(7)
      acc84(18)=Qspk2*acc84(6)
      acc84(17)=acc84(18)+acc84(3)+acc84(17)
      acc84(17)=Qspk2*acc84(17)
      acc84(18)=-acc84(10)*Qspvak2l4
      acc84(18)=acc84(18)+acc84(14)
      acc84(18)=Qspval4k1*acc84(18)
      acc84(19)=-acc84(15)*Qspvak2l5
      acc84(19)=acc84(19)+acc84(11)
      acc84(19)=Qspval5k2*acc84(19)
      acc84(20)=acc84(13)*Qspval4l5
      acc84(21)=acc84(8)*Qspl4
      acc84(22)=Qspvak1k2*acc84(2)
      acc84(23)=Qspvak2l4*acc84(9)
      acc84(24)=Qspvak2l5*acc84(12)
      acc84(25)=QspQ*acc84(5)
      brack=acc84(16)+acc84(17)+acc84(18)+acc84(19)+acc84(20)+acc84(21)+acc84(2&
      &2)+acc84(23)+acc84(24)+acc84(25)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p0_ubaru_httbar_d84h10l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p0_ubaru_httbar_globalsl1_qp, only: epspow
      use p0_ubaru_httbar_kinematics_qp
      use p0_ubaru_httbar_abbrevd84h10_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d84
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k2
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d84 = 0.0_ki
      d84 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d84, ki), aimag(d84), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p0_ubaru_httbar_d84h10l1_qp
