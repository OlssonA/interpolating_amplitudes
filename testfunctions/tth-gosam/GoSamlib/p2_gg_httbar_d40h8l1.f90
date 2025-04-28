module     p2_gg_httbar_d40h8l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity8d40h8l1.f90
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
      use p2_gg_httbar_abbrevd40h8
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc40(47)
      complex(ki) :: Qspvae2e1
      complex(ki) :: Qspvae1e2
      complex(ki) :: Qspvae2l5
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspvae2l4
      complex(ki) :: Qspval4e2
      complex(ki) :: Qspvae1l4
      complex(ki) :: Qspval4e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvak2e2
      complex(ki) :: Qspvak2e1
      complex(ki) :: Qspvae2k1
      complex(ki) :: Qspvak1e2
      complex(ki) :: Qspval5l4
      complex(ki) :: Qspval4l5
      complex(ki) :: Qspval4k2
      complex(ki) :: Qspval4k1
      complex(ki) :: Qspvak2l5
      complex(ki) :: Qspvak2l4
      complex(ki) :: Qspvak2k1
      complex(ki) :: Qspvak1l4
      complex(ki) :: Qspl4
      complex(ki) :: Qspk2
      Qspvae2e1 = dotproduct(Q,spvae2e1)
      Qspvae1e2 = dotproduct(Q,spvae1e2)
      Qspvae2l5 = dotproduct(Q,spvae2l5)
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspvae2l4 = dotproduct(Q,spvae2l4)
      Qspval4e2 = dotproduct(Q,spval4e2)
      Qspvae1l4 = dotproduct(Q,spvae1l4)
      Qspval4e1 = dotproduct(Q,spval4e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvak2e2 = dotproduct(Q,spvak2e2)
      Qspvak2e1 = dotproduct(Q,spvak2e1)
      Qspvae2k1 = dotproduct(Q,spvae2k1)
      Qspvak1e2 = dotproduct(Q,spvak1e2)
      Qspval5l4 = dotproduct(Q,spval5l4)
      Qspval4l5 = dotproduct(Q,spval4l5)
      Qspval4k2 = dotproduct(Q,spval4k2)
      Qspval4k1 = dotproduct(Q,spval4k1)
      Qspvak2l5 = dotproduct(Q,spvak2l5)
      Qspvak2l4 = dotproduct(Q,spvak2l4)
      Qspvak2k1 = dotproduct(Q,spvak2k1)
      Qspvak1l4 = dotproduct(Q,spvak1l4)
      Qspl4 = dotproduct(Q,l4)
      Qspk2 = dotproduct(Q,k2)
      acc40(1)=abb40(14)
      acc40(2)=abb40(15)
      acc40(3)=abb40(16)
      acc40(4)=abb40(17)
      acc40(5)=abb40(18)
      acc40(6)=abb40(19)
      acc40(7)=abb40(20)
      acc40(8)=abb40(21)
      acc40(9)=abb40(22)
      acc40(10)=abb40(23)
      acc40(11)=abb40(24)
      acc40(12)=abb40(26)
      acc40(13)=abb40(30)
      acc40(14)=abb40(31)
      acc40(15)=abb40(32)
      acc40(16)=abb40(34)
      acc40(17)=abb40(35)
      acc40(18)=abb40(36)
      acc40(19)=abb40(38)
      acc40(20)=abb40(40)
      acc40(21)=abb40(43)
      acc40(22)=abb40(44)
      acc40(23)=abb40(45)
      acc40(24)=abb40(54)
      acc40(25)=Qspvae2e1*acc40(5)
      acc40(26)=Qspvae1e2*acc40(11)
      acc40(27)=Qspvae2l5*acc40(19)
      acc40(28)=Qspval5e2*acc40(22)
      acc40(29)=Qspvae2l4*acc40(23)
      acc40(30)=Qspval4e2*acc40(8)
      acc40(31)=Qspvae1l4*acc40(24)
      acc40(32)=Qspval4e1*acc40(4)
      acc40(33)=Qspvae2k2*acc40(20)
      acc40(34)=Qspvak2e2*acc40(21)
      acc40(35)=Qspvak2e1*acc40(2)
      acc40(36)=Qspvae2k1*acc40(14)
      acc40(37)=Qspvak1e2*acc40(9)
      acc40(38)=Qspval5l4*acc40(17)
      acc40(39)=Qspval4l5*acc40(18)
      acc40(40)=Qspval4k2*acc40(6)
      acc40(41)=Qspval4k1*acc40(13)
      acc40(42)=Qspvak2l5*acc40(16)
      acc40(43)=Qspvak2l4*acc40(10)
      acc40(44)=Qspvak2k1*acc40(12)
      acc40(45)=Qspvak1l4*acc40(7)
      acc40(46)=Qspl4*acc40(15)
      acc40(47)=Qspk2*acc40(3)
      brack=acc40(1)+acc40(25)+acc40(26)+acc40(27)+acc40(28)+acc40(29)+acc40(30&
      &)+acc40(31)+acc40(32)+acc40(33)+acc40(34)+acc40(35)+acc40(36)+acc40(37)+&
      &acc40(38)+acc40(39)+acc40(40)+acc40(41)+acc40(42)+acc40(43)+acc40(44)+ac&
      &c40(45)+acc40(46)+acc40(47)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d40h8l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd40h8
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d40
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(-Q_ext(0:3),  ki_nin), aimag(-Q_ext(0:3)), ki)
      d40 = 0.0_ki
      d40 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d40, ki), aimag(d40), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d40h8l1
