module     p2_gg_httbar_d12h4l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d12h4l1_qp.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki => ki_qp
   use p2_gg_httbar_util_qp, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model_qp
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_color_qp
      use p2_gg_httbar_abbrevd12h4_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc12(29)
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak2l4
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspk2 = dotproduct(Q,k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      acc12(1)=abb12(9)
      acc12(2)=abb12(10)
      acc12(3)=abb12(11)
      acc12(4)=abb12(12)
      acc12(5)=abb12(13)
      acc12(6)=abb12(14)
      acc12(7)=abb12(15)
      acc12(8)=abb12(16)
      acc12(9)=abb12(17)
      acc12(10)=abb12(18)
      acc12(11)=abb12(19)
      acc12(12)=abb12(20)
      acc12(13)=abb12(21)
      acc12(14)=abb12(22)
      acc12(15)=abb12(23)
      acc12(16)=abb12(24)
      acc12(17)=abb12(25)
      acc12(18)=abb12(30)
      acc12(19)=-Qspvak2l3*acc12(18)
      acc12(20)=Qspk2*acc12(2)
      acc12(19)=acc12(20)+acc12(3)+acc12(19)
      acc12(19)=Qspk2*acc12(19)
      acc12(20)=-acc12(13)*Qspval3k1
      acc12(21)=-acc12(4)*Qspval5k1
      acc12(20)=acc12(21)+acc12(14)+acc12(20)
      acc12(20)=Qspvak1l4*acc12(20)
      acc12(21)=Qspvak1k2*acc12(1)
      acc12(22)=Qspvak1l3*acc12(15)
      acc12(21)=acc12(22)+acc12(5)+acc12(21)
      acc12(21)=Qspvak2k1*acc12(21)
      acc12(22)=acc12(13)*Qspval3k2
      acc12(23)=acc12(4)*Qspval5k2
      acc12(22)=acc12(23)+acc12(17)+acc12(22)
      acc12(22)=Qspvak2l4*acc12(22)
      acc12(23)=Qspvak1k2*acc12(10)
      acc12(24)=Qspvak1l3*acc12(16)
      acc12(25)=Qspvak2l3*acc12(8)
      acc12(26)=Qspval3k1*acc12(7)
      acc12(27)=Qspval3k2*acc12(6)
      acc12(28)=Qspval5k1*acc12(11)
      acc12(29)=Qspval5k2*acc12(9)
      brack=acc12(12)+acc12(19)+acc12(20)+acc12(21)+acc12(22)+acc12(23)+acc12(2&
      &4)+acc12(25)+acc12(26)+acc12(27)+acc12(28)+acc12(29)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d12h4l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd12h4_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d12
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k4
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d12 = 0.0_ki
      d12 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d12, ki), aimag(d12), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d12h4l1_qp
