module     p2_gg_httbar_d148h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d148h8l1.f90
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
      use p2_gg_httbar_abbrevd148h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc148(60)
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspe2
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: QspQ
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspe2 = dotproduct(Q,e2)
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      QspQ = dotproduct(Q,Q)
      acc148(1)=abb148(11)
      acc148(2)=abb148(12)
      acc148(3)=abb148(13)
      acc148(4)=abb148(14)
      acc148(5)=abb148(15)
      acc148(6)=abb148(16)
      acc148(7)=abb148(17)
      acc148(8)=abb148(18)
      acc148(9)=abb148(19)
      acc148(10)=abb148(20)
      acc148(11)=abb148(21)
      acc148(12)=abb148(22)
      acc148(13)=abb148(23)
      acc148(14)=abb148(25)
      acc148(15)=abb148(28)
      acc148(16)=abb148(32)
      acc148(17)=abb148(33)
      acc148(18)=abb148(35)
      acc148(19)=abb148(37)
      acc148(20)=abb148(40)
      acc148(21)=abb148(43)
      acc148(22)=abb148(44)
      acc148(23)=abb148(46)
      acc148(24)=abb148(47)
      acc148(25)=abb148(50)
      acc148(26)=abb148(57)
      acc148(27)=abb148(70)
      acc148(28)=abb148(74)
      acc148(29)=abb148(82)
      acc148(30)=abb148(88)
      acc148(31)=abb148(92)
      acc148(32)=abb148(96)
      acc148(33)=abb148(100)
      acc148(34)=abb148(122)
      acc148(35)=abb148(124)
      acc148(36)=acc148(2)*Qspval4e1
      acc148(37)=acc148(4)*Qspval4k2
      acc148(38)=acc148(7)*Qspval4k1
      acc148(39)=acc148(25)*Qspval4l5
      acc148(40)=acc148(26)*Qspl4
      acc148(41)=acc148(29)*Qspk2
      acc148(42)=Qspvae1k2*acc148(18)
      acc148(43)=Qspval5k2*acc148(23)
      acc148(44)=Qspvak1k2*acc148(12)
      acc148(36)=acc148(44)+acc148(43)+acc148(42)+acc148(41)+acc148(40)+acc148(&
      &39)+acc148(17)+acc148(38)+acc148(37)+acc148(36)
      acc148(36)=Qspe2*acc148(36)
      acc148(37)=acc148(1)*Qspval4k2
      acc148(38)=acc148(5)*Qspk2
      acc148(39)=acc148(6)*Qspval4k1
      acc148(40)=acc148(15)*Qspl4
      acc148(41)=acc148(19)*Qspval4e1
      acc148(42)=acc148(24)*Qspval4l5
      acc148(43)=Qspvae2e1*acc148(33)
      acc148(44)=Qspvae1e2*acc148(28)
      acc148(45)=Qspvae2l5*acc148(34)
      acc148(46)=Qspval5e2*acc148(27)
      acc148(47)=Qspvae2l4*acc148(32)
      acc148(48)=Qspval4e2*acc148(35)
      acc148(49)=Qspvae1l4*acc148(31)
      acc148(50)=Qspvae2k2*acc148(3)
      acc148(51)=Qspvak2e2*acc148(13)
      acc148(52)=Qspvak2e1*acc148(14)
      acc148(53)=Qspvae2k1*acc148(20)
      acc148(54)=Qspvak1e2*acc148(22)
      acc148(55)=Qspval5l4*acc148(21)
      acc148(56)=Qspvak2l5*acc148(10)
      acc148(57)=Qspvak2l4*acc148(9)
      acc148(58)=Qspvak2k1*acc148(8)
      acc148(59)=Qspvak1l4*acc148(11)
      acc148(60)=-QspQ*acc148(30)
      brack=acc148(16)+acc148(36)+acc148(37)+acc148(38)+acc148(39)+acc148(40)+a&
      &cc148(41)+acc148(42)+acc148(43)+acc148(44)+acc148(45)+acc148(46)+acc148(&
      &47)+acc148(48)+acc148(49)+acc148(50)+acc148(51)+acc148(52)+acc148(53)+ac&
      &c148(54)+acc148(55)+acc148(56)+acc148(57)+acc148(58)+acc148(59)+acc148(6&
      &0)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d148h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd148h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d148
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d148 = 0.0_ki
      d148 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d148, ki), aimag(d148), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d148h8l1
