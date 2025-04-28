module     p2_gg_httbar_d26h8l1_qp
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d26h8l1_qp.f90
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
      use p2_gg_httbar_abbrevd26h8_qp
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc26(73)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4l3
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3l4
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspk2
      complex(ki) :: QspQ
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4l3 = dotproduct(Q,spval4l3)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3l4 = dotproduct(Q,spval3l4)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspk2 = dotproduct(Q,k2)
      QspQ = dotproduct(Q,Q)
      acc26(1)=abb26(11)
      acc26(2)=abb26(12)
      acc26(3)=abb26(13)
      acc26(4)=abb26(14)
      acc26(5)=abb26(15)
      acc26(6)=abb26(16)
      acc26(7)=abb26(17)
      acc26(8)=abb26(18)
      acc26(9)=abb26(19)
      acc26(10)=abb26(20)
      acc26(11)=abb26(21)
      acc26(12)=abb26(22)
      acc26(13)=abb26(23)
      acc26(14)=abb26(24)
      acc26(15)=abb26(25)
      acc26(16)=abb26(26)
      acc26(17)=abb26(27)
      acc26(18)=abb26(28)
      acc26(19)=abb26(29)
      acc26(20)=abb26(30)
      acc26(21)=abb26(31)
      acc26(22)=abb26(32)
      acc26(23)=abb26(33)
      acc26(24)=abb26(34)
      acc26(25)=abb26(35)
      acc26(26)=abb26(37)
      acc26(27)=abb26(39)
      acc26(28)=abb26(40)
      acc26(29)=abb26(42)
      acc26(30)=abb26(43)
      acc26(31)=abb26(44)
      acc26(32)=abb26(45)
      acc26(33)=abb26(49)
      acc26(34)=abb26(52)
      acc26(35)=abb26(57)
      acc26(36)=abb26(72)
      acc26(37)=abb26(73)
      acc26(38)=-Qspvae2e1*acc26(34)
      acc26(39)=Qspvae1e2*acc26(12)
      acc26(40)=Qspvae1l5*acc26(13)
      acc26(41)=Qspval5e1*acc26(16)
      acc26(42)=Qspvae2l4*acc26(20)
      acc26(43)=Qspval4e2*acc26(25)
      acc26(44)=Qspvae2l3*acc26(27)
      acc26(45)=Qspval3e2*acc26(29)
      acc26(46)=-Qspvae1l3*acc26(37)
      acc26(47)=Qspval3e1*acc26(35)
      acc26(48)=Qspvae2k2*acc26(30)
      acc26(49)=Qspvak2e2*acc26(1)
      acc26(50)=Qspvae1k2*acc26(2)
      acc26(51)=Qspvak2e1*acc26(36)
      acc26(52)=Qspvae2k1*acc26(9)
      acc26(53)=Qspvak1e2*acc26(21)
      acc26(54)=Qspval5l4*acc26(22)
      acc26(55)=Qspval5l3*acc26(24)
      acc26(56)=Qspval5k2*acc26(28)
      acc26(57)=Qspval5k1*acc26(26)
      acc26(58)=Qspval4l5*acc26(31)
      acc26(59)=Qspval4l3*acc26(32)
      acc26(60)=Qspval4k2*acc26(23)
      acc26(61)=Qspval3l5*acc26(33)
      acc26(62)=Qspval3l4*acc26(10)
      acc26(63)=Qspval3k2*acc26(5)
      acc26(64)=Qspval3k1*acc26(8)
      acc26(65)=Qspvak2l5*acc26(6)
      acc26(66)=Qspvak2l4*acc26(11)
      acc26(67)=Qspvak2l3*acc26(14)
      acc26(68)=Qspvak2k1*acc26(15)
      acc26(69)=Qspvak1l5*acc26(17)
      acc26(70)=Qspvak1l3*acc26(19)
      acc26(71)=Qspvak1k2*acc26(18)
      acc26(72)=Qspk2*acc26(4)
      acc26(73)=QspQ*acc26(7)
      brack=acc26(3)+acc26(38)+acc26(39)+acc26(40)+acc26(41)+acc26(42)+acc26(43&
      &)+acc26(44)+acc26(45)+acc26(46)+acc26(47)+acc26(48)+acc26(49)+acc26(50)+&
      &acc26(51)+acc26(52)+acc26(53)+acc26(54)+acc26(55)+acc26(56)+acc26(57)+ac&
      &c26(58)+acc26(59)+acc26(60)+acc26(61)+acc26(62)+acc26(63)+acc26(64)+acc2&
      &6(65)+acc26(66)+acc26(67)+acc26(68)+acc26(69)+acc26(70)+acc26(71)+acc26(&
      &72)+acc26(73)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d26h8l1_qp_ninja")
      use iso_c_binding, only: c_int
      use quadninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1_qp, only: epspow
      use p2_gg_httbar_kinematics_qp
      use p2_gg_httbar_abbrevd26h8_qp
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d26
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = k3+k5
      Q(1:4)  =cmplx(real(+Q_ext(0:3)  -qshift(:),  ki_nin), aimag(+Q_ext(0:3))&
      &, ki)
      d26 = 0.0_ki
      d26 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d26, ki), aimag(d26), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d26h8l1_qp
