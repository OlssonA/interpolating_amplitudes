module     p2_gg_httbar_d37h0l1
   ! file: /itp/swift/jannisl/fast/POWHEG-BOX-V2/ttH_for_samplecpp_updated/GoSa &
   ! &m_POWHEG/Virtual/p2_gg_httbar/helicity0d37h0l1.f90
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
      use p2_gg_httbar_abbrevd37h0
      implicit none
      complex(ki), dimension(4), intent(in) :: Q
      complex(ki), intent(in) :: mu2
      complex(ki) :: brack
      complex(ki) :: acc37(41)
      complex(ki) :: Qspval5e2
      complex(ki) :: Qspval5e1
      complex(ki) :: Qspvae2l3
      complex(ki) :: Qspval3e2
      complex(ki) :: Qspvae1l3
      complex(ki) :: Qspval3e1
      complex(ki) :: Qspvae2k2
      complex(ki) :: Qspvae1k2
      complex(ki) :: Qspval5l3
      complex(ki) :: Qspval5k2
      complex(ki) :: Qspval5k1
      complex(ki) :: Qspval3l5
      complex(ki) :: Qspval3k2
      complex(ki) :: Qspval3k1
      complex(ki) :: Qspvak2l3
      complex(ki) :: Qspvak1l3
      complex(ki) :: Qspvak1k2
      complex(ki) :: Qspl5
      complex(ki) :: Qspl3
      complex(ki) :: Qspk2
      Qspval5e2 = dotproduct(Q,spval5e2)
      Qspval5e1 = dotproduct(Q,spval5e1)
      Qspvae2l3 = dotproduct(Q,spvae2l3)
      Qspval3e2 = dotproduct(Q,spval3e2)
      Qspvae1l3 = dotproduct(Q,spvae1l3)
      Qspval3e1 = dotproduct(Q,spval3e1)
      Qspvae2k2 = dotproduct(Q,spvae2k2)
      Qspvae1k2 = dotproduct(Q,spvae1k2)
      Qspval5l3 = dotproduct(Q,spval5l3)
      Qspval5k2 = dotproduct(Q,spval5k2)
      Qspval5k1 = dotproduct(Q,spval5k1)
      Qspval3l5 = dotproduct(Q,spval3l5)
      Qspval3k2 = dotproduct(Q,spval3k2)
      Qspval3k1 = dotproduct(Q,spval3k1)
      Qspvak2l3 = dotproduct(Q,spvak2l3)
      Qspvak1l3 = dotproduct(Q,spvak1l3)
      Qspvak1k2 = dotproduct(Q,spvak1k2)
      Qspl5 = dotproduct(Q,l5)
      Qspl3 = dotproduct(Q,l3)
      Qspk2 = dotproduct(Q,k2)
      acc37(1)=abb37(15)
      acc37(2)=abb37(16)
      acc37(3)=abb37(18)
      acc37(4)=abb37(19)
      acc37(5)=abb37(20)
      acc37(6)=abb37(21)
      acc37(7)=abb37(22)
      acc37(8)=abb37(23)
      acc37(9)=abb37(24)
      acc37(10)=abb37(25)
      acc37(11)=abb37(26)
      acc37(12)=abb37(27)
      acc37(13)=abb37(28)
      acc37(14)=abb37(30)
      acc37(15)=abb37(31)
      acc37(16)=abb37(33)
      acc37(17)=abb37(34)
      acc37(18)=abb37(35)
      acc37(19)=abb37(38)
      acc37(20)=abb37(39)
      acc37(21)=abb37(41)
      acc37(22)=Qspval5e2*acc37(9)
      acc37(23)=Qspval5e1*acc37(11)
      acc37(24)=Qspvae2l3*acc37(14)
      acc37(25)=Qspval3e2*acc37(19)
      acc37(26)=Qspvae1l3*acc37(21)
      acc37(27)=Qspval3e1*acc37(20)
      acc37(28)=Qspvae2k2*acc37(10)
      acc37(29)=Qspvae1k2*acc37(17)
      acc37(30)=Qspval5l3*acc37(18)
      acc37(31)=Qspval5k2*acc37(1)
      acc37(32)=Qspval5k1*acc37(2)
      acc37(33)=Qspval3l5*acc37(5)
      acc37(34)=Qspval3k2*acc37(6)
      acc37(35)=Qspval3k1*acc37(7)
      acc37(36)=Qspvak2l3*acc37(8)
      acc37(37)=Qspvak1l3*acc37(15)
      acc37(38)=Qspvak1k2*acc37(12)
      acc37(39)=Qspl5*acc37(16)
      acc37(40)=Qspl3*acc37(13)
      acc37(41)=Qspk2*acc37(4)
      brack=acc37(3)+acc37(22)+acc37(23)+acc37(24)+acc37(25)+acc37(26)+acc37(27&
      &)+acc37(28)+acc37(29)+acc37(30)+acc37(31)+acc37(32)+acc37(33)+acc37(34)+&
      &acc37(35)+acc37(36)+acc37(37)+acc37(38)+acc37(39)+acc37(40)+acc37(41)
   end  function brack_1
!---#] function brack_1:
!---#[ numerator interfaces:
   !------#[ subroutine numerator_ninja:
   subroutine numerator_ninja(ncut, Q_ext, mu2_ext, numerator) &
   & bind(c, name="p2_gg_httbar_d37h0l1_ninja")
      use iso_c_binding, only: c_int
      use ninjago_module, only: ki_nin
      use p2_gg_httbar_globalsl1, only: epspow
      use p2_gg_httbar_kinematics
      use p2_gg_httbar_abbrevd37h0
      implicit none
      integer(c_int), intent(in) :: ncut
      complex(ki_nin), dimension(0:3), intent(in) :: Q_ext
      complex(ki_nin), intent(in) :: mu2_ext
      complex(ki_nin), intent(out) :: numerator
      complex(ki) :: d37
      ! The Q that goes into the diagram
      complex(ki), dimension(4) :: Q
      complex(ki) :: mu2
      Q(1:4)  =cmplx(real(+Q_ext(0:3),  ki_nin), aimag(+Q_ext(0:3)), ki)
      d37 = 0.0_ki
      d37 = (cond(epspow.eq.0,brack_1,Q,mu2))
      numerator = cmplx(real(d37, ki), aimag(d37), ki_nin)
   end subroutine numerator_ninja
   !------#] subroutine numerator_ninja:
!---#] numerator interfaces:
end module p2_gg_httbar_d37h0l1
