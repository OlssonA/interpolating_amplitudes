module     p2_gg_httbar_d42h4l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity4d42h4l1.f90
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
      use p2_gg_httbar_abbrevd42h4
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc42(59)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae1l5
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspvae1k1
      complex(ki) :: Qspvak1e1
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l5
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspl5
      complex(ki) :: Qspk2
      complex(ki) :: Qspk1
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae1l5 = dotproduct(Q,spvae1l5)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspvae1k1 = dotproduct(Q,spvae1k1)
      Qspvak1e1 = dotproduct(Q,spvak1e1)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l5 = dotproduct(Q,spvak1l5)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspl5 = dotproduct(Q,l5)
      Qspk2 = dotproduct(Q,k2)
      Qspk1 = dotproduct(Q,k1)
      acc42(1)=abb42(14)
      acc42(2)=abb42(15)
      acc42(3)=abb42(16)
      acc42(4)=abb42(18)
      acc42(5)=abb42(19)
      acc42(6)=abb42(21)
      acc42(7)=abb42(22)
      acc42(8)=abb42(23)
      acc42(9)=abb42(24)
      acc42(10)=abb42(26)
      acc42(11)=abb42(28)
      acc42(12)=abb42(29)
      acc42(13)=abb42(30)
      acc42(14)=abb42(31)
      acc42(15)=abb42(34)
      acc42(16)=abb42(35)
      acc42(17)=abb42(36)
      acc42(18)=abb42(37)
      acc42(19)=abb42(40)
      acc42(20)=abb42(45)
      acc42(21)=abb42(46)
      acc42(22)=abb42(49)
      acc42(23)=abb42(50)
      acc42(24)=abb42(55)
      acc42(25)=abb42(57)
      acc42(26)=abb42(59)
      acc42(27)=abb42(60)
      acc42(28)=abb42(62)
      acc42(29)=abb42(65)
      acc42(30)=abb42(67)
      acc42(31)=Qspvae2e1*acc42(14)
      acc42(32)=Qspvae1e2*acc42(16)
      acc42(33)=Qspvae2l5*acc42(10)
      acc42(34)=Qspval5e2*acc42(15)
      acc42(35)=Qspvae1l5*acc42(23)
      acc42(36)=Qspval5e1*acc42(9)
      acc42(37)=Qspvae1l4*acc42(26)
      acc42(38)=Qspval4e1*acc42(28)
      acc42(39)=Qspvak2e2*acc42(11)
      acc42(40)=Qspvae1k2*acc42(20)
      acc42(41)=Qspvak2e1*acc42(29)
      acc42(42)=Qspvae2k1*acc42(30)
      acc42(43)=Qspvak1e2*acc42(7)
      acc42(44)=Qspvae1k1*acc42(19)
      acc42(45)=Qspvak1e1*acc42(21)
      acc42(46)=Qspval5l4*acc42(24)
      acc42(47)=Qspval5k2*acc42(25)
      acc42(48)=Qspval5k1*acc42(13)
      acc42(49)=Qspval4l5*acc42(18)
      acc42(50)=Qspval4k1*acc42(3)
      acc42(51)=Qspvak2l5*acc42(22)
      acc42(52)=Qspvak2l4*acc42(27)
      acc42(53)=Qspvak2k1*acc42(2)
      acc42(54)=Qspvak1l5*acc42(4)
      acc42(55)=Qspvak1l4*acc42(12)
      acc42(56)=Qspvak1k2*acc42(5)
      acc42(57)=Qspl5*acc42(17)
      acc42(58)=Qspk2*acc42(6)
      acc42(59)=Qspk1*acc42(8)
      brack=acc42(1)+acc42(31)+acc42(32)+acc42(33)+acc42(34)+acc42(35)+acc42(36&
      &)+acc42(37)+acc42(38)+acc42(39)+acc42(40)+acc42(41)+acc42(42)+acc42(43)+&
      &acc42(44)+acc42(45)+acc42(46)+acc42(47)+acc42(48)+acc42(49)+acc42(50)+ac&
      &c42(51)+acc42(52)+acc42(53)+acc42(54)+acc42(55)+acc42(56)+acc42(57)+acc4&
      &2(58)+acc42(59)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d42h4l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd42h4
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d42
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d42 = 0.0_ki
      d42 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d42, ki), aimag(d42), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d42h4l1
