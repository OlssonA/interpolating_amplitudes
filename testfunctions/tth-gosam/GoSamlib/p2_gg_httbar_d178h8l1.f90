module     p2_gg_httbar_d178h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d178h8l1.f90
   ! generator: buildfortran.py
   use p2_gg_httbar_config, only: ki
   use p2_gg_httbar_util, only: cond
   implicit none
   private
   complex(ki), parameter :: i_ = (0.0_ki, 1.0_ki)
   public :: numerator_ninja
contains
!---#[ function brack_1:
   pure function brack_1(Q,mu2) result(brack)
      use p2_gg_httbar_model
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_color
      use p2_gg_httbar_abbrevd178h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc178(60)
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspe1
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: QspQ
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspe1 = dotproduct(Q,e1)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      QspQ = dotproduct(Q,Q)
      acc178(1)=abb178(11)
      acc178(2)=abb178(12)
      acc178(3)=abb178(13)
      acc178(4)=abb178(14)
      acc178(5)=abb178(15)
      acc178(6)=abb178(16)
      acc178(7)=abb178(17)
      acc178(8)=abb178(18)
      acc178(9)=abb178(19)
      acc178(10)=abb178(20)
      acc178(11)=abb178(21)
      acc178(12)=abb178(22)
      acc178(13)=abb178(23)
      acc178(14)=abb178(25)
      acc178(15)=abb178(27)
      acc178(16)=abb178(28)
      acc178(17)=abb178(29)
      acc178(18)=abb178(30)
      acc178(19)=abb178(31)
      acc178(20)=abb178(32)
      acc178(21)=abb178(37)
      acc178(22)=abb178(44)
      acc178(23)=abb178(45)
      acc178(24)=abb178(46)
      acc178(25)=abb178(48)
      acc178(26)=abb178(50)
      acc178(27)=abb178(51)
      acc178(28)=abb178(71)
      acc178(29)=abb178(72)
      acc178(30)=abb178(77)
      acc178(31)=abb178(82)
      acc178(32)=abb178(88)
      acc178(33)=abb178(89)
      acc178(34)=abb178(90)
      acc178(35)=abb178(107)
      acc178(36)=acc178(4)*Qspval4k2
      acc178(37)=acc178(13)*Qspval4k1
      acc178(38)=acc178(19)*Qspval4l5
      acc178(39)=acc178(22)*Qspval4e2
      acc178(40)=acc178(23)*Qspl4
      acc178(41)=acc178(28)*Qspk2
      acc178(42)=Qspvae2k2*acc178(25)
      acc178(43)=Qspval5k2*acc178(14)
      acc178(44)=Qspvak1k2*acc178(5)
      acc178(36)=acc178(44)+acc178(43)+acc178(42)+acc178(41)+acc178(40)+acc178(&
      &39)+acc178(38)+acc178(37)+acc178(7)+acc178(36)
      acc178(36)=Qspe1*acc178(36)
      acc178(37)=acc178(3)*Qspval4k2
      acc178(38)=acc178(6)*Qspk2
      acc178(39)=acc178(10)*Qspval4e2
      acc178(40)=acc178(11)*Qspval4k1
      acc178(41)=acc178(16)*Qspval4l5
      acc178(42)=acc178(18)*Qspl4
      acc178(43)=Qspvae2e1*acc178(21)
      acc178(44)=Qspvae1e2*acc178(31)
      acc178(45)=Qspvae1l5*acc178(32)
      acc178(46)=Qspval5e1*acc178(33)
      acc178(47)=Qspvae2l4*acc178(34)
      acc178(48)=Qspvae1l4*acc178(26)
      acc178(49)=Qspval4e1*acc178(24)
      acc178(50)=Qspvak2e2*acc178(1)
      acc178(51)=Qspvae1k2*acc178(27)
      acc178(52)=Qspvak2e1*acc178(29)
      acc178(53)=Qspvae1k1*acc178(2)
      acc178(54)=Qspvak1e1*acc178(8)
      acc178(55)=-Qspval5l4*acc178(35)
      acc178(56)=Qspvak2l5*acc178(15)
      acc178(57)=Qspvak2l4*acc178(20)
      acc178(58)=Qspvak2k1*acc178(9)
      acc178(59)=Qspvak1l4*acc178(12)
      acc178(60)=-QspQ*acc178(30)
      brack=acc178(17)+acc178(36)+acc178(37)+acc178(38)+acc178(39)+acc178(40)+a&
      &cc178(41)+acc178(42)+acc178(43)+acc178(44)+acc178(45)+acc178(46)+acc178(&
      &47)+acc178(48)+acc178(49)+acc178(50)+acc178(51)+acc178(52)+acc178(53)+ac&
      &c178(54)+acc178(55)+acc178(56)+acc178(57)+acc178(58)+acc178(59)+acc178(6&
      &0)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d178h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd178h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d178
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      real(ki), dimension(0:3) :: qshift
      qshift = -k3-k4-k5
      Q(1:4)  =cmplx(real(-Q_ext(0:3)  -qshift(:),  ki_nin), aimag(-Q_ext(0:3))&
      &, ki)
      d178 = 0.0_ki
      d178 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d178, ki), aimag(d178), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d178h8l1
