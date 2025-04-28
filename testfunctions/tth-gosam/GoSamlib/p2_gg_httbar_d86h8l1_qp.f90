module     p2_gg_httbar_d86h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d86h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd86h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc86(77)
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspk2
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspl5
      complex(ki) :: QspQ
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspe1
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspk2 = dotproduct(Q,k2)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspl5 = dotproduct(Q,l5)
      QspQ = dotproduct(Q,Q)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspe1 = dotproduct(Q,e1)
      acc86(1)=abb86(8)
      acc86(2)=abb86(9)
      acc86(3)=abb86(10)
      acc86(4)=abb86(11)
      acc86(5)=abb86(12)
      acc86(6)=abb86(13)
      acc86(7)=abb86(14)
      acc86(8)=abb86(15)
      acc86(9)=abb86(16)
      acc86(10)=abb86(17)
      acc86(11)=abb86(18)
      acc86(12)=abb86(19)
      acc86(13)=abb86(20)
      acc86(14)=abb86(21)
      acc86(15)=abb86(22)
      acc86(16)=abb86(23)
      acc86(17)=abb86(24)
      acc86(18)=abb86(25)
      acc86(19)=abb86(26)
      acc86(20)=abb86(27)
      acc86(21)=abb86(28)
      acc86(22)=abb86(29)
      acc86(23)=abb86(30)
      acc86(24)=abb86(31)
      acc86(25)=abb86(32)
      acc86(26)=abb86(33)
      acc86(27)=abb86(34)
      acc86(28)=abb86(35)
      acc86(29)=abb86(36)
      acc86(30)=abb86(37)
      acc86(31)=abb86(39)
      acc86(32)=abb86(40)
      acc86(33)=abb86(43)
      acc86(34)=abb86(44)
      acc86(35)=abb86(46)
      acc86(36)=abb86(48)
      acc86(37)=abb86(49)
      acc86(38)=abb86(52)
      acc86(39)=abb86(53)
      acc86(40)=abb86(54)
      acc86(41)=abb86(55)
      acc86(42)=abb86(56)
      acc86(43)=abb86(59)
      acc86(44)=abb86(61)
      acc86(45)=abb86(65)
      acc86(46)=abb86(68)
      acc86(47)=abb86(69)
      acc86(48)=abb86(70)
      acc86(49)=abb86(72)
      acc86(50)=abb86(77)
      acc86(51)=abb86(78)
      acc86(52)=abb86(80)
      acc86(53)=abb86(81)
      acc86(54)=abb86(83)
      acc86(55)=abb86(90)
      acc86(56)=acc86(11)*Qspvak2l5
      acc86(57)=acc86(14)*Qspvak1e2
      acc86(58)=acc86(21)*Qspval5l3
      acc86(59)=acc86(24)*Qspvak2e2
      acc86(60)=acc86(28)*Qspvak2l3
      acc86(61)=acc86(29)*Qspval5k2
      acc86(62)=-acc86(35)*Qspk2
      acc86(63)=acc86(38)*Qspval3e2
      acc86(64)=acc86(42)*Qspl5
      acc86(65)=acc86(45)*QspQ
      acc86(66)=acc86(49)*Qspval4e2
      acc86(56)=acc86(66)+acc86(65)+acc86(64)+acc86(63)+acc86(62)+acc86(61)+acc&
      &86(60)+acc86(27)+acc86(59)+acc86(58)+acc86(57)+acc86(56)
      acc86(56)=Qspvae2e1*acc86(56)
      acc86(57)=acc86(10)*Qspval4k2
      acc86(58)=acc86(17)*Qspvae2k1
      acc86(59)=acc86(19)*Qspvak2l5
      acc86(60)=acc86(33)*Qspval4l5
      acc86(61)=acc86(34)*Qspval3l5
      acc86(62)=acc86(41)*Qspval3k2
      acc86(63)=acc86(43)*Qspvae2l5
      acc86(64)=acc86(51)*QspQ
      acc86(65)=acc86(52)*Qspvae2l3
      acc86(66)=acc86(54)*Qspvae2k2
      acc86(67)=-acc86(55)*Qspk2
      acc86(57)=acc86(67)+acc86(66)+acc86(65)+acc86(64)+acc86(63)+acc86(62)+acc&
      &86(61)+acc86(60)+acc86(22)+acc86(59)+acc86(58)+acc86(57)
      acc86(57)=Qspvae1e2*acc86(57)
      acc86(58)=acc86(30)*Qspvae2l3
      acc86(59)=acc86(37)*Qspvae2k2
      acc86(58)=acc86(59)+acc86(58)+acc86(15)
      acc86(58)=acc86(58)*Qspvak2e2
      acc86(59)=acc86(48)*Qspval3e2
      acc86(60)=acc86(50)*Qspval4e2
      acc86(59)=acc86(60)+acc86(59)+acc86(36)
      acc86(59)=acc86(59)*Qspvae2l5
      acc86(60)=acc86(3)*Qspvae2k2
      acc86(61)=acc86(4)*Qspval4e2
      acc86(62)=acc86(7)*Qspvae2l3
      acc86(63)=acc86(40)*Qspval3e2
      acc86(58)=acc86(63)+acc86(62)+acc86(5)+acc86(61)+acc86(60)+acc86(59)+acc8&
      &6(58)
      acc86(58)=Qspe1*acc86(58)
      acc86(59)=acc86(1)*Qspval4k2
      acc86(60)=acc86(2)*Qspvak2e2
      acc86(61)=acc86(6)*Qspk2
      acc86(62)=acc86(8)*QspQ
      acc86(63)=acc86(9)*Qspvak1e2
      acc86(64)=acc86(12)*Qspvak2l5
      acc86(65)=acc86(13)*Qspl5
      acc86(66)=acc86(16)*Qspvae2k1
      acc86(67)=acc86(18)*Qspval5l3
      acc86(68)=acc86(20)*Qspval3l5
      acc86(69)=acc86(25)*Qspvak2l3
      acc86(70)=acc86(26)*Qspval5k2
      acc86(71)=acc86(31)*Qspvae2k2
      acc86(72)=acc86(32)*Qspval4l5
      acc86(73)=acc86(39)*Qspval3k2
      acc86(74)=acc86(44)*Qspvae2l3
      acc86(75)=acc86(46)*Qspval4e2
      acc86(76)=acc86(47)*Qspvae2l5
      acc86(77)=acc86(53)*Qspval3e2
      brack=acc86(23)+acc86(56)+acc86(57)+acc86(58)+acc86(59)+acc86(60)+acc86(6&
      &1)+acc86(62)+acc86(63)+acc86(64)+acc86(65)+acc86(66)+acc86(67)+acc86(68)&
      &+acc86(69)+acc86(70)+acc86(71)+acc86(72)+acc86(73)+acc86(74)+acc86(75)+a&
      &cc86(76)+acc86(77)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d86h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd86h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d86
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k2+k3+k4+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d86 = 0.0_ki
      d86 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d86, ki), aimag(d86), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d86h8l1_qp
